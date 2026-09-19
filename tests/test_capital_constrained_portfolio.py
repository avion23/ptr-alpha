"""Canaries for the capital-constrained portfolio evaluation of consensus signals.

The simulator under test is the accepted shared-cash/position-ledger
PortfolioSimulator. These canaries pin the frozen harness contract:

* one shared cash ledger and one position ledger; the accounting identity
  cash == initial + realized proceeds - open costs holds on every snapshot;
* equal funding: every entry targets 1/max_positions of total value;
* exact next-session execution: end-of-day signals never execute same-day;
* no overlap compounding: a held/pending ticker is never re-entered;
* valuation gaps abstain from risk metrics (no fictional zero mark);
* the benchmark is the real SPY column from the same price frame.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyzer import analysis
from analyzer.database import Database
from analyzer.portfolio_sim import PortfolioConfig, PortfolioSimulator

MEMBERS = ["ALICE", "BOB", "CAROL"]
TICKERS = ["AAA", "BBB", "CCC"]
PORT_START = date(2023, 7, 1)
PORT_END = date(2024, 11, 1)

GRID = {
    "horizon": [60],
    "frequency_days": [30],
    "top_n": [5],
}


def build_fixture_db(
    tmp_path: Path,
    *,
    drift: float = 0.001,
    spy_drift: float = 0.0,
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    tx_start: date = date(2021, 11, 1),
    tx_end: date = date(2024, 10, 20),
    price_end: str = "2025-01-20",
) -> Path:
    """Build a temp DuckDB whose consensus strategy trades on schedule.

    Every member buys every ticker on a staggered cadence, so on each
    scheduled rebalance at least two members have a recent disclosure for
    each ticker. Tickers rise monotonically while SPY is flat.
    """
    db_path = tmp_path / "fixture.duckdb"
    db = Database(db_path)
    dates = pd.bdate_range("2021-10-01", price_end)
    n = len(dates)
    prices = {"SPY": 100.0 * np.cumprod(1 + spy_drift * np.ones(n))}
    for ticker in tickers:
        prices[ticker] = 100.0 * np.cumprod(1 + drift * np.ones(n))
    db.upsert_prices(pd.DataFrame(prices, index=dates))

    rows = []
    doc = 0
    day = tx_start
    while day <= tx_end:
        for i, member in enumerate(MEMBERS):
            buy_date = day + timedelta(days=7 * i)
            if buy_date > tx_end:
                continue
            for ticker in tickers:
                rows.append(
                    {
                        "doc_id": f"doc-{doc:06d}",
                        "member": member,
                        "ticker": ticker,
                        "transaction_date": buy_date,
                        "disclosure_date": buy_date,
                        "transaction_type": "Purchase",
                        "owner_code": "DC",
                        "amount_midpoint": 50000.0,
                        "instrument_type": "stock",
                        "asset_description": "[ST] Common Stock",
                        "ticker_origin": "official",
                        "amount_raw": "$50,001 - $100,000",
                    }
                )
                doc += 1
        day += timedelta(days=21)
    db.upsert_transactions(pd.DataFrame(rows), source="senate_efd")
    db.close()
    return db_path


def _portfolio_config(**overrides) -> PortfolioConfig:
    values = dict(
        initial_capital=20000.0,
        max_positions=5,
        rebalance_freq_days=30,
        hold_period_days=120,
        entry_slippage_pct=0.001,
        exit_slippage_pct=0.001,
    )
    values.update(overrides)
    return PortfolioConfig(**values)


def _fixture_recommendations(db_path: Path) -> pd.DataFrame:
    """Collect consensus recommendations on the 30-day grid, as validation does."""
    db = Database(db_path, read_only=True)
    try:
        all_tx = db.get_transactions_by_date_range(
            pd.Timestamp("2021-10-07"), pd.Timestamp(PORT_END)
        )
    finally:
        db.conn.close()
    rows = []
    for as_of in pd.date_range(PORT_START, PORT_END, freq="30D"):
        recs = analysis.backtest_recommendations(
            all_tx,
            as_of_date=as_of,
            lookback_days=28,
            min_buyers=2,
            top_n=5,
        )
        if recs.empty:
            continue
        recs["as_of_date"] = as_of
        rows.append(recs)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _run_sim(recs, prices, **overrides):
    sim = PortfolioSimulator(_portfolio_config(**overrides))
    results = sim.run(recs, prices, PORT_START, PORT_END)
    return sim, results


class TestSharedLedgerAndEqualFunding:
    def test_cash_position_ledger_accounting_identity(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            prices = db.get_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2025-01-20"),
            )
        finally:
            db.conn.close()
        sim, results = _run_sim(_fixture_recommendations(db_path), prices)
        assert not results.empty
        assert (results["cash"] >= 0).all()
        # total_value is exactly cash + positions_value on every snapshot.
        assert results["total_value"].equals(
            results["cash"] + results["positions_value"]
        )
        # Cash moves only by buys/sells: cash == initial - open cost + proceeds.
        expected_cash = (
            20000.0
            + sum(cp["proceeds"] for cp in sim.closed_positions)
            - (
                sum(p.cost for p in sim.positions)
                + sum(cp["cost"] for cp in sim.closed_positions)
            )
        )
        assert results.iloc[-1]["cash"] == pytest.approx(expected_cash, abs=1e-6)
        assert results.iloc[-1]["realized_pnl"] == pytest.approx(
            sum(cp["pnl"] for cp in sim.closed_positions), abs=1e-6
        )

    def test_equal_funding_targets_one_over_max_positions(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            prices = db.get_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2025-01-20"),
            )
        finally:
            db.conn.close()
        sim, _ = _run_sim(_fixture_recommendations(db_path), prices)
        executed = [p for p in sim.positions] + [
            {
                "entry_date": cp["entry_date"],
                "entry_notional": cp["entry_notional"],
                "shares": cp["shares"],
            }
            for cp in sim.closed_positions
        ]
        assert executed, "fixture must produce at least one executed entry"
        for pos in sim.positions:
            # 1/max_positions = 20% of total value at the execution date.
            snapshot = next(
                s for s in sim.snapshots if s.date == pos.entry_date
            )
            target = 0.2 * snapshot.total_value
            assert pos.entry_notional == pytest.approx(target, rel=0.02)


class TestExecutionAndOverlap:
    def test_exact_next_session_execution_never_same_day(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            prices = db.get_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2025-01-20"),
            )
        finally:
            db.conn.close()
        recs = _fixture_recommendations(db_path)
        sim, _ = _run_sim(recs, prices)
        for pos in sim.positions:
            assert pos.entry_date > pos.signal_date
        for cp in sim.closed_positions:
            assert cp["entry_date"] > cp["signal_date"]
        # The entry date is the first tradable session strictly after the
        # end-of-day signal: no earlier business day lies strictly between.
        sessions = pd.bdate_range(PORT_START, PORT_END)
        for pos in sim.positions:
            between = [
                d
                for d in sessions
                if pos.signal_date < d.date() < pos.entry_date
            ]
            assert not between, (
                f"{pos.ticker}: entry {pos.entry_date} is not the next session "
                f"after signal {pos.signal_date}"
            )
        # At least one fixture signal lands on a weekend (30-day grid from a
        # Saturday start); its execution must be the next session (Monday),
        # never the non-trading signal day itself.
        weekend_signals = [
            (r["as_of_date"].date(), r["ticker"])
            for _, r in recs.iterrows()
            if r["as_of_date"].dayofweek >= 5
        ]
        assert weekend_signals, "fixture grid must contain weekend signals"
        signal, ticker = weekend_signals[0]
        executed = [
            p for p in sim.positions if p.ticker == ticker and p.signal_date == signal
        ] + [
            cp for cp in sim.closed_positions
            if cp["ticker"] == ticker and cp["signal_date"] == signal
        ]
        assert executed, f"weekend signal {signal} {ticker} must have executed"
        entry_date = (
            executed[0].entry_date
            if hasattr(executed[0], "entry_date")
            else executed[0]["entry_date"]
        )
        assert entry_date > signal
        assert pd.Timestamp(entry_date).dayofweek < 5
        assert (entry_date - signal).days >= 1
        assert all(
            p.execution.date.date() > PORT_END for p in sim.pending_entries
        ) or not sim.pending_entries

    def test_held_ticker_is_never_reentered_no_overlap_compounding(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            prices = db.get_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2025-01-20"),
            )
        finally:
            db.conn.close()
        sim, _ = _run_sim(_fixture_recommendations(db_path), prices)
        assert any(
            r["reason"] == "already_held_or_pending"
            for r in sim.rejected_orders
        ), "repeated signals for a held ticker must be refused"
        # A ticker is never held twice concurrently (no overlap compounding).
        for ticker in TICKERS:
            open_intervals = [
                (p.entry_date, p.entry_date + timedelta(days=120))
                for p in sim.positions
                if p.ticker == ticker
            ] + [
                (cp["entry_date"], cp["exit_date"])
                for cp in sim.closed_positions
                if cp["ticker"] == ticker
            ]
            for i, left in enumerate(open_intervals):
                for right in open_intervals[i + 1 :]:
                    assert left[1] <= right[0] or right[1] <= left[0], (
                        f"{ticker} intervals overlap: {left} vs {right}"
                    )


class TestValuationGapAndBenchmark:
    def test_valuation_gap_abstains_from_risk_metrics(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            prices = db.get_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2025-01-20"),
            )
        finally:
            db.conn.close()
        recs = _fixture_recommendations(db_path)
        gapped = prices.copy()
        gapped.loc[
            pd.Timestamp("2024-09-30"):pd.Timestamp("2024-10-20"), "AAA"
        ] = np.nan
        sim, _ = _run_sim(recs, gapped)
        assert sim.valuation_unavailable_dates, "fixture must open a position during the gap"
        metrics = sim.compute_metrics(gapped)
        assert metrics["valuation_gap_count"] > 0
        assert metrics["daily_risk_status"] == "unavailable_nonconsecutive_valuations"
        # Risk metrics abstain (None) instead of fabricating values.
        assert metrics["sharpe_ratio"] is None
        assert metrics["max_drawdown_pct"] is None
        assert metrics["volatility_pct"] is None
        assert metrics["return_status"] == "terminal_observed_after_valuation_gaps"

    def test_real_spy_benchmark_matches_manual_buy_hold(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            prices = db.get_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2025-01-20"),
            )
        finally:
            db.conn.close()
        sim, _ = _run_sim(_fixture_recommendations(db_path), prices)
        metrics = sim.compute_metrics(prices)
        assert metrics["spy_benchmark_status"] == "available"
        assert metrics["spy_return_pct"] is not None

        spy = prices["SPY"].dropna()
        entry_price = spy.loc[spy.index >= pd.Timestamp(PORT_START)].iloc[0]
        exit_price = spy.loc[spy.index <= pd.Timestamp(PORT_END)].iloc[-1]
        expected = (
            exit_price * (1 - 0.001) / (entry_price * (1 + 0.001)) - 1
        ) * 100
        assert metrics["spy_return_pct"] == pytest.approx(round(expected, 2), abs=1e-9)
