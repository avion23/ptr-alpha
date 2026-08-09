"""Deterministic Kelly sizing and shared-capital portfolio canaries."""

from datetime import date

import pandas as pd
import pytest

from analyzer.portfolio import (
    KellyConfig,
    build_kelly_portfolio,
    compute_portfolio_metrics,
    simulate_portfolio_returns,
)
from analyzer.portfolio.simulation import _attach_sizing_inputs, _prepare_sizing_inputs


def _sizing_rows(tickers, members=None):
    n = len(tickers)
    return pd.DataFrame(
        {
            "ticker": tickers,
            "member": members or [f"member-{i}" for i in range(n)],
            "signal_score": list(range(n, 0, -1)),
            "win_rate": [0.60] * n,
            "avg_win_pct": [1.50] * n,
            "avg_loss_pct": [1.20] * n,
            "crash_prob": [0.0] * n,
        }
    )


def test_half_kelly_retains_absolute_fourteen_percent_bankroll():
    # p=.60, b=1.5/1.2=1.25 -> full Kelly .28 -> half Kelly .14.
    portfolio = build_kelly_portfolio(
        _sizing_rows(["A"]),
        KellyConfig(capital=100, max_ticker_pct=1, max_member_pct=1),
    )

    assert portfolio.iloc[0]["kelly_fraction"] == pytest.approx(0.14)
    assert portfolio.iloc[0]["weight"] == pytest.approx(0.14)
    assert portfolio.iloc[0]["position_value"] == pytest.approx(14.0)


def test_missing_or_placeholder_outcome_inputs_abstain():
    assert build_kelly_portfolio(pd.DataFrame({"ticker": ["A"]})).empty
    rows = _sizing_rows(["A"])
    rows.loc[0, "member"] = "unknown"
    assert build_kelly_portfolio(rows).empty
    rows = _sizing_rows(["A"])
    rows.loc[0, "avg_loss_pct"] = float("nan")
    assert build_kelly_portfolio(rows).empty


def test_correlated_member_cap_reduces_without_redistribution():
    rows = _sizing_rows(["A", "B"], members=["same", "same"])
    portfolio = build_kelly_portfolio(
        rows,
        KellyConfig(
            capital=100, max_ticker_pct=1, max_member_pct=0.20,
            total_exposure_pct=1,
        ),
    )

    assert portfolio["kelly_fraction"].tolist() == pytest.approx([0.14, 0.14])
    assert portfolio.groupby("member")["weight"].sum().iloc[0] == pytest.approx(0.20)
    assert sorted(portfolio["weight"].tolist()) == pytest.approx([0.10, 0.10])


def test_ticker_cap_applies_to_aggregate_duplicate_ticker_exposure():
    rows = _sizing_rows(["A", "A"], members=["one", "two"])
    portfolio = build_kelly_portfolio(
        rows,
        KellyConfig(capital=100, max_ticker_pct=0.20, max_member_pct=1),
    )
    assert portfolio.groupby("ticker")["weight"].sum().iloc[0] == pytest.approx(0.20)


def test_crash_probability_is_not_applied_twice_by_default():
    rows = _sizing_rows(["A", "B"])
    rows["crash_prob"] = [0.5, 0.0]
    portfolio = build_kelly_portfolio(
        rows, KellyConfig(capital=100, max_ticker_pct=1, max_member_pct=1)
    )
    assert portfolio.set_index("ticker").loc["A", "weight"] == pytest.approx(
        portfolio.set_index("ticker").loc["B", "weight"]
    )


def test_overlapping_full_bankroll_signal_cannot_reuse_one_hundred_dollars():
    targets = pd.DataFrame(
        {
            "as_of_date": [date(2024, 1, 1), date(2024, 1, 31)],
            "ticker": ["A", "B"],
            "weight": [1.0, 1.0],
        }
    )
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 110.0, 110.0],
            "B": [None, 100.0, 100.0, 110.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-31", "2024-03-01", "2024-03-31"]),
    )

    curve = simulate_portfolio_returns(
        targets, prices, horizon=60, entry_slippage_bps=0,
        exit_slippage_bps=0, initial_capital=100,
    )
    metrics = compute_portfolio_metrics(curve)
    final = curve.iloc[-1]

    assert final["executed_positions"] == 1
    assert final["skipped_signals"] == 1
    assert final["closed_positions"] == 1
    assert final["liquidation_value"] == pytest.approx(110.0)
    assert metrics["total_return_pct"] == pytest.approx(10.0)
    assert metrics["gross_traded_notional"] == pytest.approx(210.0)


def test_open_position_reports_liquidation_cost_and_exposure():
    targets = pd.DataFrame(
        {"as_of_date": [date(2024, 1, 1)], "ticker": ["A"], "weight": [1.0]}
    )
    prices = pd.DataFrame(
        {"A": [100.0, 100.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    curve = simulate_portfolio_returns(
        targets, prices, horizon=60, entry_slippage_bps=0,
        exit_slippage_bps=100, initial_capital=100,
    )
    metrics = compute_portfolio_metrics(curve)

    assert metrics["open_positions"] == 1
    assert metrics["open_exposure"] == pytest.approx(100.0)
    assert metrics["estimated_liquidation_cost"] == pytest.approx(1.0)
    assert metrics["total_return_pct"] == pytest.approx(-1.0)
    assert metrics["close_coverage_pct"] == 0.0


def test_metrics_anchor_drawdown_and_annualize_from_actual_dates():
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-12-31"]),
            "simulation_start": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "initial_capital": [100.0, 100.0],
            "liquidation_value": [90.0, 110.0],
            "gross_traded_notional": [100.0, 200.0],
        }
    )
    metrics = compute_portfolio_metrics(curve)

    assert metrics["total_return_pct"] == pytest.approx(10.0)
    assert metrics["annualized_return_pct"] == pytest.approx(10.0, abs=0.05)
    assert metrics["max_drawdown_pct"] == pytest.approx(-10.0)
    assert metrics["elapsed_days"] == 365


def test_initial_capital_is_explicit_or_unambiguously_inferred():
    targets = pd.DataFrame(
        {"as_of_date": [date(2024, 1, 1)], "ticker": ["A"], "weight": [0.1]}
    )
    prices = pd.DataFrame({"A": [100.0]}, index=pd.to_datetime(["2024-01-01"]))
    with pytest.raises(ValueError, match="initial_capital"):
        simulate_portfolio_returns(targets, prices)


def test_empty_inputs_abstain():
    assert build_kelly_portfolio(pd.DataFrame()).empty
    assert simulate_portfolio_returns(pd.DataFrame(), pd.DataFrame()).empty
    assert compute_portfolio_metrics(pd.DataFrame()) == {}


def test_zero_mark_is_preserved_as_total_loss():
    targets = pd.DataFrame(
        {"as_of_date": [date(2024, 1, 1)], "ticker": ["A"], "weight": [1.0]}
    )
    prices = pd.DataFrame(
        {"A": [100.0, 0.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    curve = simulate_portfolio_returns(
        targets, prices, horizon=60, entry_slippage_bps=0,
        exit_slippage_bps=0, initial_capital=100,
    )
    metrics = compute_portfolio_metrics(curve)

    assert curve.iloc[-1]["open_exposure"] == 0.0
    assert curve.iloc[-1]["liquidation_value"] == 0.0
    assert metrics["total_return_pct"] == -100.0
    assert metrics["valuation_complete"] is True


def test_stale_open_mark_makes_equity_unavailable_and_blocks_new_sizing():
    targets = pd.DataFrame(
        {
            "as_of_date": [date(2024, 1, 1), date(2024, 1, 10)],
            "ticker": ["A", "B"],
            "weight": [0.5, 0.5],
        }
    )
    prices = pd.DataFrame(
        {"A": [100.0, None], "B": [10.0, 10.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-10"]),
    )

    curve = simulate_portfolio_returns(
        targets, prices, horizon=60, entry_slippage_bps=0, exit_slippage_bps=0,
        initial_capital=100, max_mark_staleness_days=3,
    )
    metrics = compute_portfolio_metrics(curve)
    final = curve.iloc[-1]

    assert pd.isna(final["liquidation_value"])
    assert final["unavailable_open_positions"] == 1
    assert final["valuation_skips"] == 1
    assert final["executed_positions"] == 1
    assert metrics["valuation_complete"] is False
    assert metrics["total_return_pct"] is None
    assert metrics["open_exposure"] is None


def test_enabled_crash_guard_abstains_on_missing_or_malformed_probability():
    rows = _sizing_rows(["A"])
    missing = rows.drop(columns="crash_prob")
    assert build_kelly_portfolio(missing, KellyConfig(crash_guard=True)).empty

    malformed = rows.copy()
    malformed.loc[0, "crash_prob"] = float("nan")
    assert build_kelly_portfolio(malformed, KellyConfig(crash_guard=True)).empty

    outside_range = rows.copy()
    outside_range.loc[0, "crash_prob"] = 1.1
    assert build_kelly_portfolio(outside_range, KellyConfig(crash_guard=True)).empty


def test_raw_invalid_targets_count_in_requested_coverage_and_skips():
    targets = pd.DataFrame(
        {
            "as_of_date": [date(2024, 1, 1)] * 3,
            "ticker": ["A", "B", ""],
            "weight": [0.1, float("nan"), 0.1],
        }
    )
    prices = pd.DataFrame(
        {"A": [100.0], "B": [100.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )

    final = simulate_portfolio_returns(
        targets, prices, initial_capital=100, entry_slippage_bps=0,
        exit_slippage_bps=0,
    ).iloc[-1]

    assert final["requested_signals"] == 3
    assert final["invalid_target_skips"] == 2
    assert final["skipped_signals"] == 2
    assert final["executed_positions"] == 1
    assert final["signal_coverage_pct"] == pytest.approx(100 / 3)


def test_all_or_skip_policy_never_counts_partial_fill_as_execution():
    targets = pd.DataFrame(
        {
            "as_of_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "ticker": ["A", "B"],
            "weight": [0.8, 0.8],
        }
    )
    prices = pd.DataFrame(
        {"A": [100.0], "B": [100.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )

    final = simulate_portfolio_returns(
        targets, prices, initial_capital=100, entry_slippage_bps=0,
        exit_slippage_bps=0,
    ).iloc[-1]

    assert final["execution_policy"] == "all_or_skip"
    assert final["executed_positions"] == 1
    assert final["cash_skips"] == 1
    assert final["partial_fills"] == 0
    assert final["requested_entry_notional"] == pytest.approx(160.0)
    assert final["filled_entry_notional"] == pytest.approx(80.0)
    assert final["notional_fill_pct"] == pytest.approx(50.0)


def test_sizing_estimate_uses_latest_date_not_after_rebalance():
    sizing = _prepare_sizing_inputs(
        pd.DataFrame(
            {
                "as_of_date": ["2023-12-01", "2024-01-01", "2024-02-01"],
                "ticker": ["A", "A", "A"],
                "member": ["m", "m", "m"],
                "win_rate": [0.55, 0.60, 0.90],
                "avg_win_pct": [1.5, 1.5, 9.0],
                "avg_loss_pct": [1.2, 1.2, 1.0],
            }
        )
    )
    recs = pd.DataFrame({"ticker": ["A"], "signal_score": [1.0]})

    attached = _attach_sizing_inputs(recs, sizing, pd.Timestamp("2024-01-15"))

    assert attached.iloc[0]["win_rate"] == pytest.approx(0.60)
    assert attached.iloc[0]["avg_win_pct"] == pytest.approx(1.5)


def test_sizing_estimate_duplicates_and_malformed_rows_fail_clearly():
    duplicate = pd.DataFrame(
        {
            "as_of_date": ["2024-01-01", "2024-01-01"],
            "ticker": ["A", "A"],
            "member": ["m", "m"],
            "win_rate": [0.6, 0.7],
            "avg_win_pct": [1.5, 1.5],
            "avg_loss_pct": [1.2, 1.2],
        }
    )
    with pytest.raises(ValueError, match="duplicate dated ticker"):
        _prepare_sizing_inputs(duplicate)

    malformed = duplicate.iloc[:1].copy()
    malformed.loc[0, "as_of_date"] = "not-a-date"
    with pytest.raises(ValueError, match="malformed rows"):
        _prepare_sizing_inputs(malformed)
