"""Portfolio evaluation with two deliberately independent contracts.

``equal_slot`` owns PortfolioConfig/PortfolioSimulator for the ``portfolio`` CLI:
whole shares, equal slots, exact-session fills/marks, explicit daily run bounds.

The public functions below support Kelly research (scripts/run_kelly_backtest.py):
``kelly`` builds weights from explicit dated sizing estimates, ``simulation``
executes fractional weighted targets, and ``metrics`` describes that curve.
Research fill-delay/mark sensitivities do not set equal-slot deployment policy.
The engines share the NYSE calendar, not sizing or valuation semantics.
"""

from analyzer.portfolio.kelly import (
    KellyConfig,
    build_kelly_portfolio,
    compute_payout_ratio,
    half_kelly,
    kelly_fraction,
)
from analyzer.portfolio.metrics import compute_portfolio_metrics
from analyzer.portfolio.simulation import (
    build_portfolios_from_backtest,
    simulate_portfolio_returns,
)

__all__ = [
    "KellyConfig",
    "kelly_fraction",
    "half_kelly",
    "compute_payout_ratio",
    "build_kelly_portfolio",
    "build_portfolios_from_backtest",
    "simulate_portfolio_returns",
    "compute_portfolio_metrics",
]
