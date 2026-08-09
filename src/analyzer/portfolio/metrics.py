"""Metrics for the dated, shared-capital Kelly equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_portfolio_metrics(equity_curve: pd.DataFrame) -> dict:
    """Compute performance, execution coverage, costs, and open exposure."""
    if equity_curve.empty:
        return {}
    required = {
        "date", "simulation_start", "initial_capital",
        "liquidation_value", "gross_traded_notional",
    }
    missing = required - set(equity_curve.columns)
    if missing:
        raise ValueError(f"equity_curve missing required columns: {sorted(missing)}")

    curve = equity_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    curve["simulation_start"] = pd.to_datetime(
        curve["simulation_start"], errors="coerce"
    )
    if curve["date"].isna().any() or curve["simulation_start"].isna().any():
        raise ValueError("date and simulation_start must contain valid dates")
    curve = curve.sort_values("date", kind="stable").reset_index(drop=True)

    try:
        capital_series = pd.to_numeric(curve["initial_capital"], errors="raise")
        value_series = pd.to_numeric(curve["liquidation_value"], errors="raise")
        gross_series = pd.to_numeric(
            curve["gross_traded_notional"], errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("required equity fields must be numeric") from exc
    if (
        not np.isfinite(capital_series).all()
        or capital_series.le(0).any()
        or not np.allclose(capital_series, capital_series.iloc[0])
    ):
        raise ValueError("initial_capital must be one positive finite value")
    if np.isinf(value_series).any() or value_series.dropna().lt(0).any():
        raise ValueError("liquidation_value must be non-negative or NaN when unavailable")
    if not np.isfinite(gross_series).all() or gross_series.lt(0).any():
        raise ValueError("gross_traded_notional must be non-negative and finite")
    if gross_series.diff().dropna().lt(-1e-9).any():
        raise ValueError("gross_traded_notional must be cumulative and non-decreasing")
    if curve["simulation_start"].nunique() != 1:
        raise ValueError("simulation_start must be constant across the equity curve")

    curve["initial_capital"] = capital_series.astype(float)
    curve["liquidation_value"] = value_series.astype(float)
    curve["gross_traded_notional"] = gross_series.astype(float)
    curve = curve.sort_values("date", kind="stable").reset_index(drop=True)
    dates = curve["date"]
    initial_date = pd.Timestamp(curve.iloc[0]["simulation_start"])
    if (dates < initial_date).any():
        raise ValueError("equity observation date cannot precede simulation_start")
    initial = float(curve.iloc[0]["initial_capital"])
    values = curve["liquidation_value"].to_numpy(float)

    elapsed_days = (dates.iloc[-1] - initial_date).days
    valuation_complete = bool(np.isfinite(values).all())
    gross_traded = float(curve.iloc[-1]["gross_traded_notional"])
    if valuation_complete:
        final = float(values[-1])
        total_return = final / initial - 1.0
        annualized_return = _annualized_return(total_return, elapsed_days)
        anchored = np.concatenate([[initial], values])
        previous = anchored[:-1]
        returns = np.divide(
            np.diff(anchored),
            previous,
            out=np.zeros_like(previous),
            where=previous > 0,
        )
        sharpe = _annualized_sharpe(returns, dates, initial_date)
        max_drawdown = _max_drawdown(values, initial)
        average_equity = float(np.mean(anchored))
        gross_turnover = gross_traded / average_equity if average_equity > 0 else 0.0
        performance = {
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(annualized_return * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "gross_turnover_rate": round(gross_turnover, 3),
        }
    else:
        performance = {
            "total_return_pct": None,
            "annualized_return_pct": None,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "gross_turnover_rate": None,
        }

    final_row = curve.iloc[-1]
    return {
        **performance,
        "valuation_complete": valuation_complete,
        "gross_traded_notional": round(gross_traded, 2),
        "requested_entry_notional": round(
            float(final_row.get("requested_entry_notional", 0.0)), 2
        ),
        "filled_entry_notional": round(
            float(final_row.get("filled_entry_notional", 0.0)), 2
        ),
        "notional_fill_pct": round(
            float(final_row.get("notional_fill_pct", 0.0)), 1
        ),
        "partial_fills": int(final_row.get("partial_fills", 0)),
        "requested_signals": int(final_row.get("requested_signals", 0)),
        "executed_positions": int(final_row.get("executed_positions", 0)),
        "closed_positions": int(final_row.get("closed_positions", 0)),
        "open_positions": int(final_row.get("open_positions", 0)),
        "unavailable_open_positions": int(
            final_row.get("unavailable_open_positions", 0)
        ),
        "valuation_coverage_pct": round(
            float(final_row.get("valuation_coverage_pct", 0.0)), 1
        ),
        "signal_coverage_pct": round(float(final_row.get("signal_coverage_pct", 0.0)), 1),
        "close_coverage_pct": round(float(final_row.get("close_coverage_pct", 0.0)), 1),
        "open_exposure": _optional_round(final_row.get("open_exposure", 0.0), 2),
        "open_exposure_pct": _optional_round(
            final_row.get("open_exposure_pct", 0.0), 2
        ),
        "estimated_liquidation_cost": _optional_round(
            final_row.get("estimated_liquidation_cost", 0.0), 2
        ),
        "realized_transaction_costs": round(
            float(final_row.get("realized_transaction_costs", 0.0)), 2
        ),
        "elapsed_days": elapsed_days,
        "n_equity_observations": len(curve),
    }


def _optional_round(value, digits: int) -> float | None:
    numeric = float(value)
    return round(numeric, digits) if np.isfinite(numeric) else None


def _annualized_return(total_return: float, elapsed_days: int) -> float:
    if elapsed_days <= 0:
        return 0.0
    growth = 1.0 + total_return
    if growth <= 0:
        return -1.0
    return growth ** (365.25 / elapsed_days) - 1.0


def _annualized_sharpe(
    returns: np.ndarray, dates: pd.Series, initial_date: pd.Timestamp
) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 0 or not np.isfinite(std):
        return 0.0
    all_dates = pd.DatetimeIndex([initial_date, *dates.tolist()])
    gaps = np.diff(all_dates.to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "D")
    positive_gaps = gaps[gaps > 0]
    if positive_gaps.size == 0:
        return 0.0
    observations_per_year = 365.25 / float(np.mean(positive_gaps))
    return float(np.mean(returns) / std * np.sqrt(observations_per_year))


def _max_drawdown(values: np.ndarray, initial: float) -> float:
    equity = np.concatenate([[initial], values])
    peak = np.maximum.accumulate(equity)
    return float(np.min((equity - peak) / peak))
