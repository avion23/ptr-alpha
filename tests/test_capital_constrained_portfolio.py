"""Canaries for the capital-constrained portfolio evaluation of consensus signals.

The simulator under test is the accepted shared-cash/position-ledger
PortfolioSimulator. These canaries pin the frozen harness contract:

* one shared cash ledger and one position ledger; the accounting identity
  cash == initial + realized proceeds - open costs holds on every snapshot;
* equal funding: every entry targets 1/max_positions of total value;
* exact next-session execution: end-of-day signals never execute same-day;
* no overlap compounding: a held/pending ticker is never re-entered;
* valuation gaps abstain from risk metrics (no fictional zero mark);
* the benchmark is the real SPY column from the same price frame;
* null canaries: ticker/date block permutations destroy the temporal
  association and fail closed, while member-label shuffles are invariant
  for consensus.
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
from analyzer.validation import sweep_configs, select_config

from tests.test_validation_harness import build_fixture_db, GRID

MEMBERS = ["ALICE", "BOB", "CAROL"]
TICKERS = ["AAA", "BBB", "CCC"]
PORT_START = date(2023, 7, 1)
PORT_END = date(2024, 11, 1)


def _portfolio_config(**overrides) -> PortfolioConfig:
    values = dict(
        initial_capital=20000.0,
        max_positions=5,
        max_position_pct=0.25,
        max_sector_pct=0.60,
        rebalance_freq_days=30,
        hold_period_days=120,
        entry_slippage_pct=0.001,
        exit_slippage_pct=0.001,
        min_signal_score=0.0,
        max_price_staleness_days=5,
        max_execution_wait_days=7,
        sector_by_ticker={ticker: "Tech" for ticker in TICKERS},
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
        prices = db.get_prices(
            ["SPY", *TICKERS],
            pd.Timestamp("2021-10-07"),
            pd.Timestamp("2025-01-20"),
        )
        entry_prices = db.get_entry_prices(
            ["SPY", *TICKERS],
            pd.Timestamp("2021-10-07"),
            pd.Timestamp("2025-01-20"),
        )
    finally:
        db.conn.close()
    signals = analysis.calculate_signal_potential(
        entry_prices, prices, [60], decay_lambda=0.005
    )
    rows = []
    for as_of in pd.date_range(PORT_START, PORT_END, freq="30D"):
        recs = analysis.backtest_recommendations(
            signals, all_tx, as_of_date=as_of, horizon=60, lookback_days=60,
            min_buyers=2, top_n=5, threshold=5.0, prices_df=prices,
            training_lookback_days=365, scoring_mode="consensus",
            bayes_prior_strength=20.0,
        )
        if recs.empty:
            continue
        recs = recs.drop(columns=["optimal_horizon"], errors="ignore")
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


def _block_permute_prices(
    prices: pd.DataFrame, *, ticker_blocks: bool, seed: int, block_days: int = 30
) -> pd.DataFrame:
    """Permute price cells in (ticker x date) blocks to destroy the signal.

    With ``ticker_blocks=False`` each ticker's own path is permuted in time
    blocks (date-block null). With ``ticker_blocks=True`` the assignment of
    price paths to ticker identities is permuted across time blocks as well
    (ticker x date block null). The calendar stays continuous; SPY is never
    permuted.
    """
    rng = np.random.default_rng(seed)
    dates = prices.index
    tickers = [c for c in prices.columns if c != "SPY"]
    blocks = [dates[i : i + block_days] for i in range(0, len(dates), block_days)]
    arrays = {col: prices[col].to_numpy() for col in tickers}
    out = prices.copy()
    if ticker_blocks:
        # Joint permutation of (date-block, ticker) cells: the association of
        # price paths to ticker identities is broken in time blocks.
        cells = [(bi, col) for bi in range(len(blocks)) for col in tickers]
        perm = rng.permutation(len(cells))
        for k, (bi, col) in enumerate(cells):
            src_bi, src_col = cells[perm[k]]
            vals = arrays[src_col][src_bi * block_days : (src_bi + 1) * block_days]
            out.loc[blocks[bi], col] = np.resize(vals, len(blocks[bi]))
    else:
        # Date-block null: each ticker's own path is permuted in time blocks.
        for col in tickers:
            values = arrays[col]
            block_values = [
                values[i * block_days : (i + 1) * block_days]
                for i in range(len(blocks))
            ]
            source_order = rng.permutation(len(blocks))
            for target_bi, source_bi in enumerate(source_order):
                out.loc[blocks[target_bi], col] = np.resize(
                    block_values[source_bi], len(blocks[target_bi])
                )
    return out


class TestNullCanaries:
    def test_date_block_permutation_null_fails_closed_despite_positive_mean(
        self, tmp_path
    ):
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            prices = db.get_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
            entry_prices = db.get_entry_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
            all_tx = db.get_transactions_by_date_range(
                pd.Timestamp("2021-10-07"), pd.Timestamp("2023-05-01")
            )
        finally:
            db.conn.close()

        base = sweep_configs(all_tx, prices, entry_prices, GRID, date(2022, 1, 1), date(2023, 5, 1))
        base_series = base.attrs["series_by_trial"][0]
        selection = select_config(base, 0.05, n_permutations=999, permutation_seed=0)
        assert selection["n_statistical_survivors"] == 1

        null_prices = _block_permute_prices(prices, ticker_blocks=False, seed=42)
        null = sweep_configs(
            all_tx, null_prices, entry_prices, GRID, date(2022, 1, 1), date(2023, 5, 1)
        )
        null_series = null.attrs["series_by_trial"][0]
        assert not base_series.equals(null_series)
        # The block null preserves the mean but destroys the temporal
        # dependence; the dependence-aware gate must refuse it.
        assert null.iloc[0]["overall_alpha"] > 0
        selection_null = select_config(
            null, 0.05, n_permutations=999, permutation_seed=0
        )
        assert selection_null["n_statistical_survivors"] == 0
        assert selection_null["failure_reason"] == "no_dependence_safe_survivor"

    def test_ticker_date_block_permutation_null_fails_closed(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            prices = db.get_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
            entry_prices = db.get_entry_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
            all_tx = db.get_transactions_by_date_range(
                pd.Timestamp("2021-10-07"), pd.Timestamp("2023-05-01")
            )
        finally:
            db.conn.close()

        base = sweep_configs(all_tx, prices, entry_prices, GRID, date(2022, 1, 1), date(2023, 5, 1))
        base_series = base.attrs["series_by_trial"][0]

        null_prices = _block_permute_prices(prices, ticker_blocks=True, seed=11)
        null = sweep_configs(
            all_tx, null_prices, entry_prices, GRID, date(2022, 1, 1), date(2023, 5, 1)
        )
        null_series = null.attrs["series_by_trial"][0]
        assert not base_series.equals(null_series)
        selection_null = select_config(
            null, 0.05, n_permutations=999, permutation_seed=0
        )
        assert selection_null["n_statistical_survivors"] == 0
        assert selection_null["failure_reason"] == "no_dependence_safe_survivor"

    def test_member_label_shuffle_is_invariant_for_consensus(self, tmp_path):
        """Consensus has no member-identity hypothesis: shuffling member
        labels leaves the per-date net-alpha series exactly unchanged."""
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            prices = db.get_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
            entry_prices = db.get_entry_prices(
                ["SPY", *TICKERS],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
            all_tx = db.get_transactions_by_date_range(
                pd.Timestamp("2021-10-07"), pd.Timestamp("2023-05-01")
            )
        finally:
            db.conn.close()

        base = sweep_configs(all_tx, prices, entry_prices, GRID, date(2022, 1, 1), date(2023, 5, 1))
        base_series = base.attrs["series_by_trial"][0]
        for shuffle in (
            {"ALICE": "BOB", "BOB": "CAROL", "CAROL": "ALICE"},
            {"ALICE": "CAROL", "BOB": "ALICE", "CAROL": "BOB"},
        ):
            tx = all_tx.copy()
            tx["member"] = tx["member"].map(shuffle)
            permuted = sweep_configs(
                tx, prices, entry_prices, GRID, date(2022, 1, 1), date(2023, 5, 1)
            )
            assert permuted.attrs["series_by_trial"][0].equals(base_series)
