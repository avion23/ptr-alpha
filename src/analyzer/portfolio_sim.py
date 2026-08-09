"""Shared-cash portfolio simulation with causal session-aligned execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Mapping

import numpy as np
import pandas as pd

from analyzer.backtest.prices import (
    AlignedPrice,
    _aligned_price_at_or_before_arrays,
    _aligned_price_on_or_after_arrays,
    _next_tradable_price_arrays,
)
from analyzer.signals import _price_arrays

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    initial_capital: float = 20000.0
    max_positions: int = 5
    max_position_pct: float = 0.25
    max_sector_pct: float = 0.40
    rebalance_freq_days: int = 14
    hold_period_days: int = 120
    entry_slippage_pct: float = 0.001
    exit_slippage_pct: float = 0.001
    min_signal_score: float = 0.0
    max_price_staleness_days: int = 5
    max_execution_wait_days: int = 7
    sector_by_ticker: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    ticker: str
    signal_date: date
    entry_date: date
    entry_price: float
    shares: int
    cost: float
    entry_notional: float
    sector: str
    signal_score: float
    rank: int


@dataclass(frozen=True, slots=True)
class PendingEntry:
    recommendation: pd.Series
    signal_date: date
    execution: AlignedPrice


@dataclass
class PortfolioSnapshot:
    date: date
    cash: float
    positions: list[PortfolioPosition]
    total_value: float
    dollar_exposure: float
    unrealized_pnl: float
    realized_pnl: float


class PortfolioValuationUnavailable(RuntimeError):
    """Raised when an open position has no bounded observable mark."""


class PortfolioSimulator:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions: list[PortfolioPosition] = []
        self.pending_entries: list[PendingEntry] = []
        self.closed_positions: list[dict] = []
        self.rejected_orders: list[dict] = []
        self.snapshots: list[PortfolioSnapshot] = []
        self.gross_traded_notional = 0.0
        self.valuation_unavailable_dates: list[date] = []
        self._scheduled_signals: set[tuple[date, str]] = set()
        self._simulation_end_date: date | None = None

    def run(
        self,
        recommendations: pd.DataFrame,
        prices_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Run daily accounting; end-of-day signals execute next session."""
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.config.rebalance_freq_days < 1:
            raise ValueError("rebalance_freq_days must be positive")

        self._simulation_end_date = end_date
        recs = recommendations.copy()
        has_as_of = "as_of_date" in recs.columns
        if has_as_of:
            recs["as_of_date"] = pd.to_datetime(recs["as_of_date"])

        current = start_date
        while current <= end_date:
            self._try_exit_expired(prices_df, current)
            self._execute_pending_entries(prices_df, current)

            is_rebalance = (
                current - start_date
            ).days % self.config.rebalance_freq_days == 0
            if is_rebalance:
                if has_as_of:
                    candidates = recs[recs["as_of_date"].dt.date == current]
                else:
                    candidates = recs
                self._schedule_entries(candidates, prices_df, current)

            try:
                self._record_snapshot(prices_df, current)
            except PortfolioValuationUnavailable:
                self.valuation_unavailable_dates.append(current)
            current += timedelta(days=1)

        return self.results()

    def _schedule_entries(
        self, candidates: pd.DataFrame, prices_df: pd.DataFrame, signal_date: date
    ) -> None:
        if candidates.empty:
            return
        for _, rec in candidates.sort_values(
            "signal_score", ascending=False
        ).iterrows():
            ticker = str(rec["ticker"])
            key = (signal_date, ticker)
            if key in self._scheduled_signals:
                continue
            self._scheduled_signals.add(key)
            if float(rec.get("signal_score", 0)) < self.config.min_signal_score:
                self._reject(ticker, signal_date, "signal_below_minimum")
                continue
            if any(p.ticker == ticker for p in self.positions) or any(
                p.recommendation["ticker"] == ticker for p in self.pending_entries
            ):
                self._reject(ticker, signal_date, "already_held_or_pending")
                continue

            arrays = _price_arrays(prices_df, ticker)
            if arrays is None or arrays[0] is None:
                self._reject(ticker, signal_date, "no_price_history")
                continue
            execution = _next_tradable_price_arrays(
                arrays[0],
                arrays[1],
                signal_date,
                max_wait_days=self.config.max_execution_wait_days,
            )
            if execution is None:
                self._reject(ticker, signal_date, "no_next_tradable_session")
                continue
            self.pending_entries.append(
                PendingEntry(rec.copy(), signal_date, execution)
            )

    def _execute_pending_entries(self, prices_df: pd.DataFrame, current: date) -> None:
        due = [p for p in self.pending_entries if p.execution.date.date() <= current]
        self.pending_entries = [
            p for p in self.pending_entries if p.execution.date.date() > current
        ]
        for order in sorted(
            due,
            key=lambda p: float(p.recommendation.get("signal_score", 0)),
            reverse=True,
        ):
            if len(self.positions) >= self.config.max_positions:
                self._reject(
                    str(order.recommendation["ticker"]), current, "max_positions"
                )
                continue
            try:
                self._try_enter(order, prices_df, current)
            except PortfolioValuationUnavailable:
                self._reject(
                    str(order.recommendation["ticker"]),
                    current,
                    "portfolio_valuation_unavailable",
                )

    def _try_enter(
        self, order: PendingEntry, prices_df: pd.DataFrame, execution_date: date
    ) -> None:
        rec = order.recommendation
        ticker = str(rec["ticker"])
        if order.execution.date.date() != execution_date:
            self._reject(ticker, execution_date, "execution_date_mismatch")
            return
        raw_price = order.execution.price
        if not np.isfinite(raw_price) or raw_price <= 0:
            self._reject(ticker, execution_date, "invalid_entry_price")
            return

        sector = self._get_sector(ticker, rec)
        total_value = self._total_value(prices_df, execution_date)
        target_pct = min(1.0 / self.config.max_positions, self.config.max_position_pct)
        target_value = total_value * target_pct

        sector_exposure = self._sector_exposure(prices_df, execution_date)
        current_sector_value = sector_exposure.get(sector, 0.0) * total_value
        sector_room = total_value * self.config.max_sector_pct - current_sector_value
        if sector_room <= 0:
            self._reject(ticker, execution_date, "sector_limit")
            return

        entry_price = raw_price * (1 + self.config.entry_slippage_pct)
        invest_amount = min(target_value, self.cash, sector_room)
        if invest_amount < entry_price:
            self._reject(ticker, execution_date, "insufficient_cash")
            return
        shares = int(invest_amount / entry_price)
        cost = shares * entry_price
        if shares <= 0 or cost > self.cash:
            self._reject(ticker, execution_date, "insufficient_cash")
            return

        self.cash -= cost
        raw_notional = shares * raw_price
        self.gross_traded_notional += raw_notional
        self.positions.append(
            PortfolioPosition(
                ticker=ticker,
                signal_date=order.signal_date,
                entry_date=execution_date,
                entry_price=entry_price,
                shares=shares,
                cost=cost,
                entry_notional=raw_notional,
                sector=sector,
                signal_score=float(rec.get("signal_score", 0)),
                rank=int(rec.get("rank", 0)),
            )
        )

    def _try_exit_expired(self, prices_df: pd.DataFrame, current: date) -> None:
        for pos in list(self.positions):
            target = pos.entry_date + timedelta(days=self.config.hold_period_days)
            if current < target:
                continue
            arrays = _price_arrays(prices_df, pos.ticker)
            if arrays is None or arrays[0] is None:
                continue
            execution = _aligned_price_on_or_after_arrays(
                arrays[0],
                arrays[1],
                target,
                max_wait_days=None,
                allow_zero=True,
            )
            if execution is None or execution.date.date() > current:
                continue
            self._try_exit(pos, execution)

    def _try_exit(self, pos: PortfolioPosition, execution: AlignedPrice) -> None:
        if pos not in self.positions:
            return
        raw_price = execution.price
        if not np.isfinite(raw_price) or raw_price < 0:
            return
        exit_price = raw_price * (1 - self.config.exit_slippage_pct)
        proceeds = pos.shares * exit_price
        raw_notional = pos.shares * raw_price
        self.gross_traded_notional += raw_notional
        self.cash += proceeds
        self.closed_positions.append(
            {
                "ticker": pos.ticker,
                "signal_date": pos.signal_date,
                "entry_date": pos.entry_date,
                "exit_date": execution.date.date(),
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "shares": pos.shares,
                "cost": pos.cost,
                "proceeds": proceeds,
                "entry_notional": pos.entry_notional,
                "exit_notional": raw_notional,
                "pnl": proceeds - pos.cost,
                "return_pct": (exit_price / pos.entry_price - 1) * 100,
                "sector": pos.sector,
                "signal_score": pos.signal_score,
                "rank": pos.rank,
                "holding_days": (execution.date.date() - pos.entry_date).days,
            }
        )
        self.positions = [p for p in self.positions if p is not pos]

    def _get_sector(self, ticker: str, rec: pd.Series | None = None) -> str:
        """Resolve only deterministic stored sector data; never make live calls."""
        sector = rec.get("sector") if rec is not None and "sector" in rec else None
        if sector is None or pd.isna(sector) or not str(sector).strip():
            sector = self.config.sector_by_ticker.get(ticker)
        if sector is None or not str(sector).strip():
            raise ValueError(
                f"Missing stored sector for {ticker}; provide recommendation.sector "
                "or PortfolioConfig.sector_by_ticker"
            )
        return str(sector)

    def _aligned_mark(
        self, ticker: str, prices_df: pd.DataFrame, as_of: date
    ) -> AlignedPrice | None:
        arrays = _price_arrays(prices_df, ticker)
        if arrays is None or arrays[0] is None:
            return None
        return _aligned_price_at_or_before_arrays(
            arrays[0],
            arrays[1],
            as_of,
            max_staleness_days=self.config.max_price_staleness_days,
            allow_zero=True,
        )

    def _position_value(
        self, pos: PortfolioPosition, prices_df: pd.DataFrame, as_of: date
    ) -> float:
        mark = self._aligned_mark(pos.ticker, prices_df, as_of)
        if mark is None:
            raise PortfolioValuationUnavailable(
                f"No bounded mark for open position {pos.ticker} on {as_of}"
            )
        # Mark at executable liquidation value so final equity includes exit cost.
        return pos.shares * mark.price * (1 - self.config.exit_slippage_pct)

    def _sector_exposure(
        self, prices_df: pd.DataFrame, as_of: date
    ) -> dict[str, float]:
        if not self.positions:
            return {}
        values: dict[str, float] = {}
        for pos in self.positions:
            values[pos.sector] = values.get(pos.sector, 0.0) + self._position_value(
                pos, prices_df, as_of
            )
        total = self._total_value(prices_df, as_of)
        return (
            {sector: value / total for sector, value in values.items()}
            if total > 0
            else {}
        )

    def _total_value(self, prices_df: pd.DataFrame, as_of: date) -> float:
        return self.cash + sum(
            self._position_value(pos, prices_df, as_of) for pos in self.positions
        )

    def _record_snapshot(self, prices_df: pd.DataFrame, as_of: date) -> None:
        exposure = sum(
            self._position_value(pos, prices_df, as_of) for pos in self.positions
        )
        total_value = self.cash + exposure
        realized = sum(cp["pnl"] for cp in self.closed_positions)
        unrealized = total_value - self.cash - sum(p.cost for p in self.positions)
        self.snapshots.append(
            PortfolioSnapshot(
                date=as_of,
                cash=self.cash,
                positions=list(self.positions),
                total_value=total_value,
                dollar_exposure=exposure,
                unrealized_pnl=unrealized,
                realized_pnl=realized,
            )
        )

    def _reject(self, ticker: str, as_of: date, reason: str) -> None:
        self.rejected_orders.append({"ticker": ticker, "date": as_of, "reason": reason})

    def results(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": snap.date,
                    "cash": snap.cash,
                    "positions_value": snap.dollar_exposure,
                    "dollar_exposure": snap.dollar_exposure,
                    "total_value": snap.total_value,
                    "num_positions": len(snap.positions),
                    "unrealized_pnl": snap.unrealized_pnl,
                    "realized_pnl": snap.realized_pnl,
                }
                for snap in self.snapshots
            ]
        )

    def compute_metrics(self, prices_df: pd.DataFrame | None = None) -> dict:
        if self.positions and prices_df is None:
            raise ValueError(
                "prices_df is required to value open positions; no fictional zero mark is allowed"
            )

        results = self.results()
        end_date = self._simulation_end_date or (
            pd.Timestamp(results.iloc[-1]["date"]).date()
            if not results.empty
            else date.today()
        )
        open_ledger = self._open_ledger(prices_df, end_date)
        unresolved = [row for row in open_ledger if row["liquidation_value"] is None]
        if unresolved:
            return self._unavailable_metrics(open_ledger, unresolved)
        if results.empty:
            return {}

        values = results["total_value"].to_numpy(dtype=float)
        dates = pd.to_datetime(results["date"])
        initial = self.config.initial_capital
        total_return = (values[-1] / initial - 1) * 100
        days = (dates.iloc[-1] - dates.iloc[0]).days
        ann_return = ((values[-1] / initial) ** (365.0 / max(days, 1)) - 1) * 100
        daily_returns = (
            np.diff(values) / values[:-1] if len(values) > 1 else np.array([0.0])
        )
        if len(daily_returns) > 1 and np.std(daily_returns, ddof=1) > 0:
            sharpe = float(
                np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        full_values = np.concatenate([[initial], values])
        peaks = np.maximum.accumulate(full_values)
        max_drawdown = float(np.min((full_values - peaks) / peaks)) * 100
        wins = sum(cp["pnl"] > 0 for cp in self.closed_positions)
        closed_count = len(self.closed_positions)
        avg_hold = (
            float(np.mean([cp["holding_days"] for cp in self.closed_positions]))
            if closed_count
            else 0.0
        )
        avg_value = float(np.mean(values)) if len(values) else initial
        turnover = self.gross_traded_notional / avg_value if avg_value > 0 else 0.0
        open_exposure = sum(float(row["liquidation_value"]) for row in open_ledger)
        spy_return, spy_reason = self._spy_buy_hold(
            prices_df, dates.iloc[0].date(), end_date
        )

        return {
            "valuation_status": "available",
            "valuation_reason": None,
            "valuation_gap_count": len(self.valuation_unavailable_dates),
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(ann_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(wins / closed_count * 100, 1)
            if closed_count
            else 0.0,
            "avg_holding_days": round(avg_hold, 1),
            "turnover_rate": round(turnover, 3),
            "gross_traded_notional": round(self.gross_traded_notional, 2),
            "open_dollar_exposure": round(open_exposure, 2),
            "open_position_count": len(open_ledger),
            "open_positions": open_ledger,
            "unresolved_expired_positions": sum(
                row["state"].startswith("exit_unresolved") for row in open_ledger
            ),
            "pending_order_count": len(self.pending_entries),
            "rejected_order_count": len(self.rejected_orders),
            "max_concurrent_positions": int(results["num_positions"].max()),
            "total_closed_trades": closed_count,
            "sector_concentration": self._sector_concentration(),
            "spy_return_pct": spy_return,
            "spy_benchmark_status": "available"
            if spy_return is not None
            else "omitted",
            "spy_benchmark_reason": spy_reason,
        }

    def _unavailable_metrics(
        self, open_ledger: list[dict], unresolved: list[dict]
    ) -> dict:
        return {
            "valuation_status": "unavailable",
            "valuation_reason": "unbounded_open_position_mark",
            "valuation_gap_count": len(self.valuation_unavailable_dates),
            "total_return_pct": None,
            "annualized_return_pct": None,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "win_rate_pct": None,
            "avg_holding_days": None,
            "turnover_rate": None,
            "gross_traded_notional": round(self.gross_traded_notional, 2),
            "open_dollar_exposure": None,
            "open_position_count": len(open_ledger),
            "open_positions": open_ledger,
            "unresolved_expired_positions": sum(
                row["state"].startswith("exit_unresolved") for row in unresolved
            ),
            "pending_order_count": len(self.pending_entries),
            "rejected_order_count": len(self.rejected_orders),
            "max_concurrent_positions": None,
            "total_closed_trades": len(self.closed_positions),
            "sector_concentration": self._sector_concentration(),
            "spy_return_pct": None,
            "spy_benchmark_status": "omitted",
            "spy_benchmark_reason": "portfolio_valuation_unavailable",
        }

    def _open_ledger(self, prices_df: pd.DataFrame | None, as_of: date) -> list[dict]:
        if self.positions and prices_df is None:
            raise ValueError(
                "prices_df is required to produce the open-position ledger"
            )
        ledger = []
        for pos in self.positions:
            mark = self._aligned_mark(pos.ticker, prices_df, as_of)
            liquidation_value = (
                pos.shares * mark.price * (1 - self.config.exit_slippage_pct)
                if mark is not None
                else None
            )
            exit_due = (as_of - pos.entry_date).days >= self.config.hold_period_days
            state = "exit_unresolved" if exit_due else "open"
            if mark is None:
                state += "_valuation_unavailable"
            ledger.append(
                {
                    "ticker": pos.ticker,
                    "signal_date": pos.signal_date,
                    "entry_date": pos.entry_date,
                    "shares": pos.shares,
                    "sector": pos.sector,
                    "cost": round(pos.cost, 2),
                    "mark_price": round(mark.price, 4) if mark is not None else None,
                    "mark_date": mark.date.date() if mark is not None else None,
                    "staleness_days": mark.staleness_days if mark is not None else None,
                    "liquidation_value": (
                        round(liquidation_value, 2)
                        if liquidation_value is not None
                        else None
                    ),
                    "state": state,
                }
            )
        return ledger

    def _sector_concentration(self) -> dict[str, float]:
        notionals: dict[str, float] = {}
        for cp in self.closed_positions:
            notionals[cp["sector"]] = (
                notionals.get(cp["sector"], 0.0) + cp["entry_notional"]
            )
        total = sum(notionals.values())
        return (
            {sector: value / total * 100 for sector, value in notionals.items()}
            if total > 0
            else {}
        )

    def _spy_buy_hold(
        self, prices_df: pd.DataFrame | None, start: date, end: date
    ) -> tuple[float | None, str | None]:
        if prices_df is None:
            return None, "spy_prices_not_supplied"
        arrays = _price_arrays(prices_df, "SPY")
        if arrays is None or arrays[0] is None:
            return None, "spy_prices_unavailable"
        entry = _aligned_price_on_or_after_arrays(
            arrays[0],
            arrays[1],
            start,
            max_wait_days=self.config.max_execution_wait_days,
        )
        exit_ = _aligned_price_at_or_before_arrays(
            arrays[0],
            arrays[1],
            end,
            max_staleness_days=self.config.max_price_staleness_days,
            allow_zero=True,
        )
        if entry is None:
            return None, "spy_entry_outside_boundary"
        if exit_ is None:
            return None, "spy_exit_outside_boundary"
        if exit_.date < entry.date:
            return None, "spy_window_inverted"
        entry_price = entry.price * (1 + self.config.entry_slippage_pct)
        exit_price = exit_.price * (1 - self.config.exit_slippage_pct)
        return round((exit_price / entry_price - 1) * 100, 2), None
