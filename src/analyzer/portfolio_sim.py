"""Portfolio-level simulation with overlapping positions and realistic constraints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    initial_capital: float = 20000.0
    max_positions: int = 5
    max_position_pct: float = 0.25  # max 25% per position
    max_sector_pct: float = 0.40  # max 40% per sector
    rebalance_freq_days: int = 14
    hold_period_days: int = 120
    entry_slippage_pct: float = 0.001  # 10bps
    exit_slippage_pct: float = 0.001
    min_signal_score: float = 0.0


@dataclass
class PortfolioPosition:
    ticker: str
    entry_date: date
    entry_price: float
    shares: int
    cost: float
    sector: str
    signal_score: float
    rank: int


@dataclass
class PortfolioSnapshot:
    date: date
    cash: float
    positions: list[PortfolioPosition]
    total_value: float
    unrealized_pnl: float
    realized_pnl: float


class PortfolioSimulator:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions: list[PortfolioPosition] = []
        self.closed_positions: list[dict] = []
        self.snapshots: list[PortfolioSnapshot] = []

    def run(
        self,
        recommendations: pd.DataFrame,
        prices_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Run portfolio simulation across all dates.

        `recommendations` is the full DataFrame from `backtest_recommendations`
        (may contain rows from multiple as_of dates, each with an `as_of_date`
        column).  If there is no `as_of_date` column the caller is expected to
        supply a single set of recommendations that applies to the whole period.

        Returns a DataFrame of daily portfolio snapshots.
        """
        has_as_of = "as_of_date" in recommendations.columns
        if has_as_of:
            recommendations = recommendations.copy()
            recommendations["as_of_date"] = pd.to_datetime(recommendations["as_of_date"])

        current = start_date
        while current <= end_date:
            self._try_exit_expired(prices_df, current)

            if has_as_of:
                as_of_candidates = recommendations[
                    recommendations["as_of_date"].dt.date == current
                ]
            else:
                as_of_candidates = recommendations

            if not as_of_candidates.empty:
                sorted_recs = as_of_candidates.sort_values(
                    "signal_score", ascending=False
                )
                for _, rec in sorted_recs.iterrows():
                    if len(self.positions) >= self.config.max_positions:
                        break
                    self._try_enter(rec, prices_df, current)

            self._record_snapshot(prices_df, current)
            current += timedelta(days=1)

        return self.results()

    def _try_enter(self, rec: pd.Series, prices_df: pd.DataFrame, as_of: date):
        """Attempt to enter a position if constraints allow."""
        ticker = rec["ticker"]

        # Skip if already holding this ticker
        if any(p.ticker == ticker for p in self.positions):
            return

        # Check min signal score
        signal_score = float(rec.get("signal_score", 0))
        if signal_score < self.config.min_signal_score:
            return

        # Get entry price with slippage
        as_of_ts = pd.Timestamp(as_of)
        if ticker not in prices_df.columns:
            return
        price_series = prices_df[ticker].dropna()
        if price_series.empty:
            return
        # Ensure index is DatetimeIndex for comparison
        if not isinstance(price_series.index, pd.DatetimeIndex):
            price_series.index = pd.to_datetime(price_series.index)
        eligible = price_series[price_series.index <= as_of_ts]
        if eligible.empty:
            return
        raw_price = float(eligible.iloc[-1])
        entry_price = raw_price * (1 + self.config.entry_slippage_pct)

        # Calculate available cash and position sizing
        total_value = self._total_value(prices_df, as_of)
        available_slots = self.config.max_positions - len(self.positions)
        if available_slots <= 0:
            return

        # Equal-weight among available slots, capped by max_position_pct
        target_pct = min(1.0 / self.config.max_positions, self.config.max_position_pct)
        target_value = total_value * target_pct

        # Sector constraint check
        sector = self._get_sector(ticker)
        sector_exposure = self._sector_exposure()
        current_sector_pct = sector_exposure.get(sector, 0.0)
        max_sector_value = total_value * self.config.max_sector_pct
        current_sector_value = current_sector_pct * total_value
        sector_room = max_sector_value - current_sector_value
        if sector_room <= 0:
            return

        invest_amount = min(target_value, self.cash, sector_room)
        if invest_amount < entry_price:
            return

        shares = int(invest_amount / entry_price)
        cost = shares * entry_price

        if cost > self.cash:
            return

        self.cash -= cost
        rank = int(rec.get("rank", 0))
        self.positions.append(
            PortfolioPosition(
                ticker=ticker,
                entry_date=as_of,
                entry_price=entry_price,
                shares=shares,
                cost=cost,
                sector=sector,
                signal_score=signal_score,
                rank=rank,
            )
        )

    def _try_exit_expired(self, prices_df: pd.DataFrame, as_of: date):
        """Exit positions that have exceeded the hold period."""
        expired = [
            p for p in self.positions
            if (as_of - p.entry_date).days >= self.config.hold_period_days
        ]
        for pos in expired:
            self._try_exit(pos.ticker, prices_df, as_of)

    def _try_exit(self, ticker: str, prices_df: pd.DataFrame, as_of: date):
        """Exit a specific position."""
        pos = next((p for p in self.positions if p.ticker == ticker), None)
        if pos is None:
            return

        as_of_ts = pd.Timestamp(as_of)
        if ticker not in prices_df.columns:
            return
        price_series = prices_df[ticker].dropna()
        if price_series.empty:
            return
        if not isinstance(price_series.index, pd.DatetimeIndex):
            price_series.index = pd.to_datetime(price_series.index)
        eligible = price_series[price_series.index <= as_of_ts]
        if eligible.empty:
            return
        exit_price = float(eligible.iloc[-1]) * (1 - self.config.exit_slippage_pct)

        proceeds = pos.shares * exit_price
        self.cash += proceeds
        self.closed_positions.append(
            {
                "ticker": pos.ticker,
                "entry_date": pos.entry_date,
                "exit_date": as_of,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "shares": pos.shares,
                "cost": pos.cost,
                "proceeds": proceeds,
                "pnl": proceeds - pos.cost,
                "return_pct": (exit_price / pos.entry_price - 1) * 100,
                "sector": pos.sector,
                "signal_score": pos.signal_score,
                "rank": pos.rank,
                "holding_days": (as_of - pos.entry_date).days,
            }
        )
        self.positions = [p for p in self.positions if p.ticker != ticker]

    def _get_sector(self, ticker: str) -> str:
        """Get sector for a ticker via yfinance, cached."""
        if not hasattr(self, "_sector_cache"):
            self._sector_cache: dict[str, str] = {}
        if ticker in self._sector_cache:
            return self._sector_cache[ticker]
        try:
            import yfinance as yf
            sector = yf.Ticker(ticker).info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"
        self._sector_cache[ticker] = sector
        return sector

    def _sector_exposure(self) -> dict[str, float]:
        """Current sector allocation as fraction of total value."""
        if not self.positions:
            return {}
        total = sum(p.cost for p in self.positions) + self.cash
        if total <= 0:
            return {}
        sector_values: dict[str, float] = {}
        for p in self.positions:
            sector_values[p.sector] = sector_values.get(p.sector, 0) + p.cost
        return {s: v / total for s, v in sector_values.items()}

    def _total_value(self, prices_df: pd.DataFrame, as_of: date) -> float:
        """Total portfolio value (cash + mark-to-market positions)."""
        value = self.cash
        as_of_ts = pd.Timestamp(as_of)
        for pos in self.positions:
            if pos.ticker not in prices_df.columns:
                value += pos.cost
                continue
            price_series = prices_df[pos.ticker].dropna()
            if price_series.empty:
                value += pos.cost
                continue
            if not isinstance(price_series.index, pd.DatetimeIndex):
                price_series.index = pd.to_datetime(price_series.index)
            eligible = price_series[price_series.index <= as_of_ts]
            if not eligible.empty:
                value += pos.shares * float(eligible.iloc[-1])
            else:
                value += pos.cost
        return value

    def _record_snapshot(self, prices_df: pd.DataFrame, as_of: date):
        total = self._total_value(prices_df, as_of)
        unrealized = total - self.cash - sum(p.cost for p in self.positions)
        realized = sum(cp["pnl"] for cp in self.closed_positions)
        self.snapshots.append(
            PortfolioSnapshot(
                date=as_of,
                cash=self.cash,
                positions=list(self.positions),
                total_value=total,
                unrealized_pnl=unrealized,
                realized_pnl=realized,
            )
        )

    def results(self) -> pd.DataFrame:
        """Return daily portfolio values as DataFrame."""
        if not self.snapshots:
            return pd.DataFrame(
                columns=[
                    "date", "cash", "total_value", "num_positions",
                    "unrealized_pnl", "realized_pnl",
                ]
            )
        rows = []
        for s in self.snapshots:
            rows.append(
                {
                    "date": s.date,
                    "cash": s.cash,
                    "total_value": s.total_value,
                    "num_positions": len(s.positions),
                    "unrealized_pnl": s.unrealized_pnl,
                    "realized_pnl": s.realized_pnl,
                }
            )
        return pd.DataFrame(rows)

    def compute_metrics(self, prices_df: pd.DataFrame | None = None) -> dict:
        """Compute portfolio performance metrics."""
        if not self.snapshots:
            return {}

        results = self.results()
        values = results["total_value"].values
        dates = pd.to_datetime(results["date"])
        initial = self.config.initial_capital

        # Total return
        total_return = (values[-1] / initial - 1) * 100

        # Annualized return
        days = (dates.iloc[-1] - dates.iloc[0]).days
        ann_return = ((values[-1] / initial) ** (365.0 / max(days, 1)) - 1) * 100

        # Daily returns
        daily_returns = np.diff(values) / values[:-1] if len(values) > 1 else np.array([0.0])

        # Sharpe ratio (annualized)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        max_drawdown = float(np.min(drawdowns)) * 100

        # Win rate
        if self.closed_positions:
            wins = sum(1 for cp in self.closed_positions if cp["pnl"] > 0)
            win_rate = wins / len(self.closed_positions) * 100
        else:
            win_rate = 0.0

        # Average holding period
        if self.closed_positions:
            avg_hold = np.mean([cp["holding_days"] for cp in self.closed_positions])
        else:
            avg_hold = 0.0

        # Turnover rate (total cost traded / average portfolio value)
        total_traded = sum(cp["cost"] for cp in self.closed_positions)
        avg_value = np.mean(values) if len(values) > 0 else initial
        turnover = total_traded / avg_value if avg_value > 0 else 0.0

        # Max concurrent positions
        max_concurrent = max(results["num_positions"]) if not results.empty else 0

        # Sector concentration
        sector_counts: dict[str, int] = {}
        for cp in self.closed_positions:
            sector_counts[cp["sector"]] = sector_counts.get(cp["sector"], 0) + 1
        sector_concentration = {
            s: c / len(self.closed_positions) * 100
            for s, c in sector_counts.items()
        } if self.closed_positions else {}

        # SPY comparison
        spy_return = None
        if prices_df is not None and "SPY" in prices_df.columns:
            spy_series = prices_df["SPY"].dropna()
            start_ts = pd.Timestamp(dates.iloc[0])
            end_ts = pd.Timestamp(dates.iloc[-1])
            spy_start = spy_series[spy_series.index <= start_ts]
            spy_end = spy_series[spy_series.index <= end_ts]
            if not spy_start.empty and not spy_end.empty:
                spy_return = (float(spy_end.iloc[-1]) / float(spy_start.iloc[-1]) - 1) * 100

        return {
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(ann_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 1),
            "avg_holding_days": round(float(avg_hold), 1),
            "turnover_rate": round(turnover, 3),
            "max_concurrent_positions": max_concurrent,
            "total_closed_trades": len(self.closed_positions),
            "sector_concentration": sector_concentration,
            "spy_return_pct": round(spy_return, 2) if spy_return is not None else None,
        }
