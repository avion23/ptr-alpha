"""
Ornstein-Uhlenbeck return process model for entry valuation.

Models each position's return r(t) = P(t)/P(entry) - 1 as:
    dr = θ(μ - r)dt + σdW

Entry value at time 0 (r=0):
    V(0) = μ/ρ + (0 - μ)/(θ + ρ)
         = μ * θ / (ρ * (θ + ρ))

V(0) > 0 → expected profitable entry
V(0) < 0 → expected loss → skip

Parameters are fit from historical return curves via AR(1) MLE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OUParams:
    """Ornstein-Uhlenbeck parameters estimated from a return curve."""

    theta: float
    mu: float
    sigma: float

    @property
    def a(self) -> float:
        """AR(1) coefficient: a = e^{-theta}."""
        return float(np.exp(-self.theta))

    @property
    def one_minus_a(self) -> float:
        return 1.0 - self.a

    @property
    def sigma2_ou(self) -> float:
        """OU innovation variance: sigma^2 / (2*theta) * (1 - e^{-2*theta})."""
        if self.theta < 1e-8:
            return float(self.sigma ** 2)
        return float(
            self.sigma ** 2 / (2.0 * self.theta) * (1.0 - np.exp(-2.0 * self.theta))
        )

    @property
    def beta(self) -> float:
        """Location parameter: beta = mu * (1 - a)."""
        return self.mu * self.one_minus_a


@dataclass(frozen=True, slots=True)
class OUPosterior:
    """Posterior over mu at a point in time."""

    mu_mean: float
    mu_var: float
    theta: float
    sigma2_ou: float
    n_obs: int

    @property
    def mu_std(self) -> float:
        return float(np.sqrt(max(self.mu_var, 0.0)))

    def v(self, r_tau: float, rho: float) -> float:
        """Expected discounted future reward V(tau)."""
        denom = self.theta + rho
        if abs(denom) < 1e-12:
            return 0.0
        return self.mu_mean / rho + (r_tau - self.mu_mean) / denom



# ---------------------------------------------------------------------------
# AR(1) / OU fitting (offline, on historical curves)
# ---------------------------------------------------------------------------

def fit_ar1(returns: np.ndarray) -> tuple[float, float, float]:
    """
    Fit AR(1): r(t+1) = b + a*r(t) + eps, eps ~ N(0, s2).

    Returns (a, b, s2).
    Optimized for small arrays (3-90 elements): avoids np.asarray overhead
    by checking dtype first, uses raw Python loops for tiny arrays.
    """
    n = len(returns)
    if n < 3:
        return 0.95, 0.0, 0.01

    # Avoid np.asarray overhead when input is already float64
    if isinstance(returns, np.ndarray) and returns.dtype == np.float64:
        r = returns
    else:
        r = np.asarray(returns, dtype=np.float64)

    # For very small arrays (n <= 10), use pure Python to avoid numpy overhead
    if n <= 10:
        r_bar = 0.0
        for i in range(n):
            r_bar += r[i]
        r_bar /= n

        denom = 0.0
        cov = 0.0
        for i in range(n - 1):
            d_prev = r[i] - r_bar
            d_next = r[i + 1] - r_bar
            denom += d_prev * d_prev
            cov += d_next * d_prev

        if denom < 1e-12:
            var_sum = 0.0
            for i in range(1, n):
                d = r[i] - r_bar
                var_sum += d * d
            return 0.95, 0.0, max(var_sum / max(n - 1, 1), 1e-10)

        a = cov / denom
        a = max(-0.99, min(a, 0.999))
        b = r_bar * (1.0 - a)
        s2 = 0.0
        for i in range(n - 1):
            resid = r[i + 1] - b - a * r[i]
            s2 += resid * resid
        s2 /= (n - 1)
        return a, b, max(s2, 1e-10)

    # Standard numpy path for larger arrays
    r_bar = float(r.mean())
    r_next = r[1:]
    r_prev = r[:-1]

    r_next_dm = r_next - r_bar
    r_prev_dm = r_prev - r_bar

    denom = float(np.dot(r_prev_dm, r_prev_dm))
    if denom < 1e-12:
        return 0.95, 0.0, float(np.var(r_next_dm))

    a = float(np.dot(r_next_dm, r_prev_dm) / denom)
    a = max(-0.99, min(a, 0.999))  # stationarity

    b = r_bar * (1.0 - a)
    resid = r_next - b - a * r_prev
    s2 = float(np.mean(resid ** 2))

    return a, b, max(s2, 1e-10)


def ar1_to_ou(a: float, b: float, s2: float) -> OUParams:
    """Convert AR(1) parameters to OU parameters. Uses math module for speed."""
    import math
    a = max(a, 1e-6)
    theta = -math.log(a)
    mu = b / (1.0 - a) if abs(1.0 - a) > 1e-8 else 0.0
    if theta > 1e-8:
        sigma2 = s2 * 2.0 * theta / (1.0 - math.exp(-2.0 * theta))
    else:
        sigma2 = s2
    sigma = math.sqrt(max(sigma2, 1e-10))
    return OUParams(theta=theta, mu=mu, sigma=sigma)


def fit_ou(returns: np.ndarray) -> OUParams:
    """Fit OU process to an observed return curve via AR(1) MLE."""
    a, b, s2 = fit_ar1(returns)
    return ar1_to_ou(a, b, s2)


# ---------------------------------------------------------------------------
# Prior construction from historical data (offline, one-time)
# ---------------------------------------------------------------------------

def build_prior(
    historical_return_curves: list[np.ndarray],
    default_theta: float = 0.05,
    default_sigma2: float = 0.01,
) -> tuple[float, float, float, float]:
    """
    Build global prior from historical return curves.

    Returns (theta_global, beta_prior, P_prior, sigma2_ou).
    """
    if not historical_return_curves:
        return default_theta, 0.0, 1.0, default_sigma2

    thetas = []
    betas = []
    s2s = []

    for curve in historical_return_curves:
        c = np.asarray(curve, dtype=np.float64)
        if len(c) < 3:
            continue
        ou = fit_ou(c)
        if 0.001 < ou.theta < 5.0:  # sane range
            thetas.append(ou.theta)
            betas.append(ou.beta)
            s2s.append(ou.sigma2_ou)

    if not betas:
        return default_theta, 0.0, 1.0, default_sigma2

    theta_global = float(np.median(thetas))
    beta_prior = float(np.mean(betas))
    sigma2_ou = float(np.mean(s2s))

    # Prior variance = cross-position variance of beta + floor
    beta_var = float(np.var(betas)) if len(betas) > 1 else 1.0
    P_prior = max(beta_var, 0.01)

    return theta_global, beta_prior, P_prior, sigma2_ou


# ---------------------------------------------------------------------------
# Entry valuation: V(0) from historical curves
# ---------------------------------------------------------------------------

def compute_entry_value(
    historical_return_curves: list[np.ndarray],
    rho: float = 0.000137,
    default_theta: float = 0.05,
    default_mu: float = 0.0,
) -> tuple[float, float, float]:
    """
    Estimate entry value V(0) from historical return curves for a ticker.

    Returns (V0, mu, theta).
    V0 = mu * theta / (rho * (theta + rho))
    """
    if not historical_return_curves:
        return default_mu * default_theta / (rho * (default_theta + rho)), default_mu, default_theta

    thetas = []
    mus = []

    for curve in historical_return_curves:
        c = np.asarray(curve, dtype=np.float64)
        if len(c) < 3:
            continue
        ou = fit_ou(c)
        if 0.001 < ou.theta < 5.0:
            thetas.append(ou.theta)
            mus.append(ou.mu)

    if not mus:
        return default_mu * default_theta / (rho * (default_theta + rho)), default_mu, default_theta

    mu = float(np.mean(mus))
    theta = float(np.median(thetas))
    v0 = mu * theta / (rho * (theta + rho))
    return v0, mu, theta


def compute_entry_value_and_horizon(
    historical_return_curves: list[np.ndarray],
    rho: float = 0.000137,
    default_theta: float = 0.05,
    default_mu: float = 0.0,
    min_horizon: int = 20,
    max_horizon: int = 120,
    default_horizon: int = 60,
) -> tuple[float, float, float, int]:
    """Compute both entry value V(0) and optimal horizon in a single pass.

    Fits OU once per curve and derives both V0 and optimal holding period.
    Returns (V0, mu, theta, optimal_horizon).
    """
    if not historical_return_curves:
        v0 = default_mu * default_theta / (rho * (default_theta + rho))
        return v0, default_mu, default_theta, default_horizon

    thetas = []
    mus = []

    for curve in historical_return_curves:
        c = np.asarray(curve, dtype=np.float64)
        if len(c) < 3:
            continue
        ou = fit_ou(c)
        if 0.001 < ou.theta < 5.0:
            thetas.append(ou.theta)
            mus.append(ou.mu)

    if not mus:
        v0 = default_mu * default_theta / (rho * (default_theta + rho))
        return v0, default_mu, default_theta, default_horizon

    mu = float(np.mean(mus))
    theta = float(np.median(thetas))
    v0 = mu * theta / (rho * (theta + rho))

    # Optimal horizon from half-life
    if theta < 1e-6:
        optimal = max_horizon
    else:
        half_life = np.log(2) / theta
        optimal = int(2 * half_life)
        optimal = max(min_horizon, min(max_horizon, optimal))

    return v0, mu, theta, optimal


# ---------------------------------------------------------------------------
# Optimal horizon from OU half-life
# ---------------------------------------------------------------------------

def compute_optimal_horizon(
    historical_return_curves: list[np.ndarray],
    min_horizon: int = 20,
    max_horizon: int = 120,
    default_horizon: int = 60,
) -> int:
    """Estimate optimal holding period from OU half-life.

    Uses 2 * half-life as the optimal exit (captures ~75% of mean reversion).
    Clamped to [min_horizon, max_horizon].
    """
    if not historical_return_curves:
        return default_horizon

    thetas = []
    for curve in historical_return_curves:
        c = np.asarray(curve, dtype=np.float64)
        if len(c) < 3:
            continue
        ou = fit_ou(c)
        if 0.001 < ou.theta < 5.0:
            thetas.append(ou.theta)

    if not thetas:
        return default_horizon

    median_theta = float(np.median(thetas))
    if median_theta < 1e-6:
        return max_horizon

    half_life = np.log(2) / median_theta
    optimal = int(2 * half_life)  # 2 half-lives captures ~75% of reversion
    return max(min_horizon, min(max_horizon, optimal))


# ---------------------------------------------------------------------------
# Kalman filter for online mu tracking
# ---------------------------------------------------------------------------

class KalmanFilter1D:
    """
    Scalar Kalman filter for a constant-plus-noise state.

    State model:  x(t+1) = x(t) + eta(t),  eta ~ N(0, Q)
    Observation:  z(t) = x(t) + eps(t),     eps ~ N(0, R)
    """

    def __init__(self, x0: float, P0: float, Q: float, R: float):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def update(self, z: float) -> tuple[float, float]:
        """Predict + correct. Returns (posterior_mean, posterior_var)."""
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred

        return self.x, self.P


# ---------------------------------------------------------------------------
# Per-position tracker (online, continuous)
# ---------------------------------------------------------------------------

class ReturnProcessTracker:
    """
    Tracks a single position's return process via Kalman filter.
    Makes continuous exit decisions via discounted future reward.

    Usage:
        tracker = ReturnProcessTracker(theta, beta_prior, P_prior, sigma2_ou)
        for day, r_t in enumerate(daily_returns):
            posterior = tracker.update(r_t)
    """

    def __init__(
        self,
        theta: float,
        beta_prior: float,
        P_prior: float,
        sigma2_ou: float,
        rho: float = 0.000137,   # 5% annual discount rate
        process_noise: float = 0.0,  # Q: mu drift (0 = constant mu, pure averaging)
        min_observations: int = 5,
    ):
        self.theta = theta
        self.sigma2_ou = sigma2_ou
        self.rho = rho
        self.min_observations = min_observations
        self.n_obs = 0
        self.returns: list[float] = []

        self.kf = KalmanFilter1D(
            x0=beta_prior,
            P0=P_prior,
            Q=process_noise,
            R=sigma2_ou,
        )

    def update(self, r_t: float) -> OUPosterior:
        """Incorporate new return observation. Return current posterior."""
        self.returns.append(r_t)
        self.n_obs = len(self.returns) - 1  # transitions = observations - 1

        if self.n_obs < 1:
            return self._posterior()

        r_prev = self.returns[-2]
        r_curr = self.returns[-1]

        # Innovation: z = r(t) - r(t-1) * e^{-theta}
        a = np.exp(-self.theta)
        z = r_curr - r_prev * a

        # Kalman filter update on beta = mu * (1 - a)
        self.kf.update(z)

        return self._posterior()

    def get_posterior(self) -> OUPosterior:
        """Get current posterior without triggering an update."""
        return self._posterior()

    def _posterior(self) -> OUPosterior:
        one_minus_a = 1.0 - np.exp(-self.theta)
        if abs(one_minus_a) < 1e-8:
            mu_mean = self.kf.x / 1e-8
            mu_var = self.kf.P / 1e-16
        else:
            mu_mean = self.kf.x / one_minus_a
            mu_var = self.kf.P / (one_minus_a ** 2)

        return OUPosterior(
            mu_mean=float(mu_mean),
            mu_var=float(max(mu_var, 0.0)),
            theta=self.theta,
            sigma2_ou=self.sigma2_ou,
            n_obs=self.n_obs,
        )

    def ready(self) -> bool:
        """True if enough observations to make a decision."""
        return self.n_obs >= self.min_observations
