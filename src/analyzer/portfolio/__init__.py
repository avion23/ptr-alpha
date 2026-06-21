"""Portfolio construction with Kelly criterion position sizing.

Public API (re-exported here so existing `from analyzer.portfolio import X`
calls keep working after the subpackage split):
  - KellyConfig           position-sizing parameters
  - kelly_fraction, half_kelly, compute_payout_ratio
  - build_kelly_portfolio  per-ticker weight assignment
  - build_portfolios_from_backtest  per-date portfolio builder
  - simulate_portfolio_returns  holding-period returns
  - compute_portfolio_metrics  aggregate performance stats

Package layout:
  - kelly.py         Kelly math + build_kelly_portfolio
  - simulation.py    build_portfolios + simulate_portfolio_returns
  - metrics.py       compute_portfolio_metrics
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
