"""End-to-end harness canaries for purged, fail-closed validation.

Every canary runs the real validation pipeline against a temp DuckDB fixture
(real schema, real transactions, real prices) rather than mocks. The accepted
validation contract under test:

* no-trade dates earn a zero cash return and keep identical SPY benchmark
  support (no date is silently dropped);
* one per-date net-alpha statistic drives inference, correction, selection,
  and verdict;
* no-survivor sweeps fail closed (descriptive-only, no fallback, no
  reservation written);
* consensus is identity-invariant and its control is recorded non-gating
  (gating=False) in the canonical ledger;
* the canonical ledger refuses overlapping evaluations exactly once;
* the test window is labeled retrospective diagnostics only and the locked
  final phase is never queried or consumed.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyzer.database import Database
from analyzer.member_ranking.buyer_scoring import CONSENSUS_SCORER_PROVENANCE
from analyzer.validation import (
    EvaluationAlreadyConsumedError,
    PRIMARY_METRIC,
    _benchmark_return,
    _canonical_ledger_path,
    _validate_ledger,
    run_validation,
)

MEMBERS = ["ALICE", "BOB", "CAROL"]
GRID = {
    "horizon": [60],
    "frequency_days": [30],
    "training_lookback_days": [365],
    "min_buyers": [2],
    "top_n": [5],
    "decay_lambda": [0.005],
    "bayes_prior_strength": [20],
    "scoring_mode": ["consensus"],
}
TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2023, 6, 30)
TEST_START = date(2023, 7, 1)
TEST_END = date(2024, 12, 31)


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
    each ticker. Tickers rise monotonically while SPY is flat, so per-date
    net alpha is strongly positive and near-constant.
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
    db.upsert_transactions(pd.DataFrame(rows), source="senate")
    db.close()
    return db_path


def run_fixture_validation(
    db_path: Path, *, train_start=TRAIN_START, train_end=TRAIN_END,
    test_start=TEST_START, test_end=TEST_END,
) -> dict:
    return run_validation(
        db_path,
        train_start, train_end,
        test_start, test_end,
        GRID,
        n_permutations=999,
        permutation_seed=0,
        alpha=0.05,
    )


def _load_prices(db_path: Path) -> pd.DataFrame:
    db = Database(db_path, read_only=True)
    try:
        return db.get_prices(
            ["SPY", "AAA", "BBB", "CCC"],
            pd.Timestamp("2021-10-07"),
            pd.Timestamp("2025-01-20"),
        )
    finally:
        db.conn.close()


class TestNoTradeCashReturns:
    def test_no_trade_dates_earn_zero_cash_return_with_benchmark_support(self, tmp_path):
        # Buys stop 2023-01-24; from 2023-03-27 onward no recent disclosure
        # exists, so the strategy must earn a zero cash return on those dates
        # while the identical SPY benchmark keeps the date in support.
        db_path = build_fixture_db(tmp_path, tx_end=date(2023, 1, 24))
        prices = _load_prices(db_path)
        out = run_fixture_validation(db_path)
        train = out["train"]
        assert train["status"] == "descriptive_only"
        assert train["no_trade_dates"] >= 2
        assert train["dates_evaluated"] == train["scheduled_dates"]
        assert train["benchmark_dates"] == train["scheduled_dates"]

        # Reconstruct the per-date series with the same calendar and verify a
        # no-trade date contributes exactly -(SPY return): zero cash return.
        from analyzer import analysis
        from analyzer.pipeline import BacktestParams
        from analyzer.validation import _backtest_core

        db = Database(db_path, read_only=True)
        try:
            all_tx = db.get_transactions_by_date_range(
                pd.Timestamp("2021-10-07"), pd.Timestamp("2023-05-01")
            )
            entry_prices = db.get_entry_prices(
                ["SPY", "AAA", "BBB", "CCC"],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
        finally:
            db.conn.close()
        signals = analysis.calculate_signal_potential(
            entry_prices, prices, [60], decay_lambda=0.005
        )
        params = BacktestParams(
            start_date=date(2022, 1, 1),
            end_date=date(2023, 5, 1),
            horizon=60,
            lookback_days=60,
            training_lookback_days=365,
            min_buyers=2,
            top_n=5,
            frequency_days=30,
        )
        _, per_date = _backtest_core(
            all_tx, prices, params, signals, 20.0, 0.005, "consensus"
        )
        no_trade_dates = [
            as_of
            for as_of in pd.date_range("2022-01-01", "2023-05-01", freq="30D")
            if pd.Timestamp(as_of) > pd.Timestamp("2023-03-26")
        ]
        assert no_trade_dates, "fixture must contain at least one no-trade date"
        for as_of in no_trade_dates:
            spy = _benchmark_return(prices, pd.Timestamp(as_of), 60)
            assert spy is not None
            # Zero cash return: alpha on a no-trade date is exactly -(SPY).
            assert per_date.loc[pd.Timestamp(as_of)] == pytest.approx(-spy, abs=1e-9)


class TestIdenticalSpySupportAndPerDateAlpha:
    def test_identical_spy_support_and_per_date_net_alpha(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        out = run_fixture_validation(db_path)
        train = out["train"]
        assert train["dates_evaluated"] == train["scheduled_dates"]
        assert train["benchmark_dates"] == train["scheduled_dates"]
        assert train["coverage_pct"] == 100.0
        assert train["no_trade_dates"] == 0
        assert out["primary_metric"] == PRIMARY_METRIC == "mean_per_date_net_alpha"

        # The primary statistic is the mean of the per-date net-alpha series.
        prices = _load_prices(db_path)
        db = Database(db_path, read_only=True)
        try:
            all_tx = db.get_transactions_by_date_range(
                pd.Timestamp("2021-10-07"), pd.Timestamp("2023-05-01")
            )
            entry_prices = db.get_entry_prices(
                ["SPY", "AAA", "BBB", "CCC"],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
        finally:
            db.conn.close()
        from analyzer import analysis
        from analyzer.pipeline import BacktestParams
        from analyzer.validation import _backtest_core

        signals = analysis.calculate_signal_potential(
            entry_prices, prices, [60], decay_lambda=0.005
        )
        params = BacktestParams(
            start_date=date(2022, 1, 1),
            end_date=date(2023, 5, 1),
            horizon=60,
            lookback_days=60,
            training_lookback_days=365,
            min_buyers=2,
            top_n=5,
            frequency_days=30,
        )
        result, per_date = _backtest_core(
            all_tx, prices, params, signals, 20.0, 0.005, "consensus"
        )
        assert len(per_date) == train["dates_evaluated"]
        assert result.overall_alpha == pytest.approx(float(per_date.mean()), abs=1e-4)
        assert result.overall_alpha > 0
        assert result.scoring_mode == "consensus"
        assert result.scorer_provenance == CONSENSUS_SCORER_PROVENANCE

        # Spot-check one date: alpha == strategy return - SPY return exactly.
        as_of = pd.Timestamp("2022-01-01")
        spy = _benchmark_return(prices, as_of, 60)
        recs = analysis.backtest_recommendations(
            signals, all_tx, as_of_date=as_of, horizon=60, lookback_days=60,
            min_buyers=2, top_n=5, threshold=5.0, prices_df=prices,
            training_lookback_days=365, scoring_mode="consensus",
            bayes_prior_strength=20.0,
        ).drop(columns=["optimal_horizon"], errors="ignore")
        evaluated = analysis.evaluate_backtest(recs, prices, as_of, 60)
        strategy_return = float(
            pd.to_numeric(evaluated["bt_return_pct"], errors="coerce")
            .fillna(0.0)
            .sum()
            / len(recs)
        )
        assert per_date.loc[as_of] == pytest.approx(strategy_return - spy, abs=1e-9)


class TestNoSurvivorFailClosed:
    def test_flat_market_has_no_survivor_and_no_fallback(self, tmp_path):
        db_path = build_fixture_db(tmp_path, drift=0.0)
        out = run_fixture_validation(db_path)
        assert out["status"] == "no_deployable_config"
        assert out["verdict"] == "not_robust"
        assert out["selected_config"] is None
        assert out["test"]["status"] == "not_run_without_corrected_train_survivor"
        assert out["train"]["status"] == "descriptive_only"
        assert out["train"]["label"] == "not_selected_for_deployment"
        assert out["correction"]["failure_reason"] == "no_dependence_safe_survivor"
        assert out["correction"]["n_statistical_survivors"] == 0
        # A no-survivor sweep never writes a consumption reservation.
        ledger_path = _canonical_ledger_path(db_path)
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            assert [e["event_type"] for e in ledger["events"]] == []


class TestConsensusIdentityInvariance:
    def test_consensus_identity_control_is_recorded_non_gating(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        out = run_fixture_validation(db_path)
        assert out["status"] == "retrospective_positive_result"
        assert out["selected_config"]["scoring_mode"] == "consensus"
        control = out["correction"]["member_identity_control"]
        assert control["gating"] is False
        assert control["status"] == "identity_invariant"
        assert control["method"].startswith("identity_invariant_by_consensus_scorer_contract")
        assert control["requested_permutations"] == 0

        ledger = json.loads(_canonical_ledger_path(db_path).read_text())
        events = {e["event_type"] for e in ledger["events"]}
        assert {"member_identity_control", "reservation", "completion"} <= events
        identity = next(
            e for e in ledger["events"] if e["event_type"] == "member_identity_control"
        )
        assert identity["control"]["gating"] is False
        assert identity["control"]["status"] == "identity_invariant"
        _validate_ledger(ledger)

    def test_member_label_permutation_is_invariant_for_consensus(self, tmp_path):
        """Swapping member identities must not change the per-date alpha series."""
        db_path = build_fixture_db(tmp_path)
        db = Database(db_path, read_only=True)
        try:
            all_tx = db.get_transactions_by_date_range(
                pd.Timestamp("2021-10-07"), pd.Timestamp("2023-05-01")
            )
            prices = db.get_prices(
                ["SPY", "AAA", "BBB", "CCC"],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
            entry_prices = db.get_entry_prices(
                ["SPY", "AAA", "BBB", "CCC"],
                pd.Timestamp("2021-10-07"),
                pd.Timestamp("2023-06-30"),
            )
        finally:
            db.conn.close()

        from analyzer.validation import sweep_configs

        base = sweep_configs(all_tx, prices, entry_prices, GRID, date(2022, 1, 1), date(2023, 5, 1))
        shuffled = all_tx.copy()
        swap = {"ALICE": "BOB", "BOB": "CAROL", "CAROL": "ALICE"}
        shuffled["member"] = shuffled["member"].map(swap)
        permuted = sweep_configs(
            shuffled, prices, entry_prices, GRID, date(2022, 1, 1), date(2023, 5, 1)
        )
        base_series = base.attrs["series_by_trial"][0]
        permuted_series = permuted.attrs["series_by_trial"][0]
        assert base_series.equals(permuted_series)
        assert float(base_series.mean()) > 0


class TestLedgerOverlapRefusal:
    def test_overlapping_evaluation_is_refused_exactly_once(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        first = run_fixture_validation(db_path)
        assert first["status"] == "retrospective_positive_result"
        ledger = json.loads(_canonical_ledger_path(db_path).read_text())
        assert [e["event_type"] for e in ledger["events"]] == [
            "member_identity_control",
            "reservation",
            "completion",
        ]
        _validate_ledger(ledger)

        with pytest.raises(EvaluationAlreadyConsumedError, match="overlaps a consumed"):
            run_fixture_validation(db_path)
        # The refused repeat never reserves again: the chain stays valid, the
        # evaluation key is unchanged, and no second reservation/completion
        # event exists (only a non-gating diagnostic may be appended).
        after = json.loads(_canonical_ledger_path(db_path).read_text())
        _validate_ledger(after)
        reservations = [
            e for e in after["events"] if e["event_type"] == "reservation"
        ]
        completions = [
            e for e in after["events"] if e["event_type"] == "completion"
        ]
        assert len(reservations) == 1
        assert len(completions) == 1
        assert (
            reservations[0]["evaluation_key"]
            == first["evaluation_ledger"]["evaluation_key"]
        )
        assert (
            completions[0]["evaluation_key"]
            == first["evaluation_ledger"]["evaluation_key"]
        )


class TestRetrospectiveNotFinalWording:
    def test_test_window_is_retrospective_only_and_locked_final_untouched(self, tmp_path):
        db_path = build_fixture_db(tmp_path)
        out = run_fixture_validation(db_path)
        assert out["status"] == "retrospective_positive_result"
        assert out["verdict"] == "not_fresh_oos_evidence"
        assert (
            out["test"]["status"]
            == "retrospective_previously_used_not_fresh_oos"
        )
        manifest = out["manifest"]
        assert (
            manifest["phases"]["test"]["evidence_class"]
            == "retrospective_previously_used_not_fresh_oos"
        )
        locked = manifest["phases"]["locked_final"]
        assert locked["status"] == "locked_not_queried_or_evaluated"
        assert locked["consumed"] is False
        assert locked["value_rows_queried"] is False

        # No profitability wording may appear in any status/verdict; any
        # fresh-OOS mention must be an explicit denial (not_fresh_oos_*).
        for key, value in out.items():
            if isinstance(value, str):
                lowered = value.lower()
                assert "profit" not in lowered, f"{key} claims profitability"
                if "fresh_oos" in lowered:
                    assert "not_fresh_oos" in lowered, f"{key} claims fresh OOS"
        assert out["degradation_ratio"] is not None
        assert "retrospective" in out["evaluation_ledger"]["status"]
