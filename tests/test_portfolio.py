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
